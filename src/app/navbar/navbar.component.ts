import { Component, OnInit,
  HostBinding, HostListener, ElementRef, QueryList, ViewChildren, ViewChild, AfterViewInit, ChangeDetectorRef} from '@angular/core';
import { NavigationStart, Router } from '@angular/router';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { MatDialog} from '@angular/material/dialog';
import { DialogComponent} from '../dialog/dialog.component'
import { GlobalService } from '../global.service';
import { BehaviorSubject } from 'rxjs';

@Component({
  selector: 'app-navbar',
  templateUrl: './navbar.component.html',
  styleUrls: ['./navbar.component.css']
})

export class NavbarComponent implements OnInit, AfterViewInit {
  isNavbarCollapsed = true;
  isMathidden = true;
  selected: string;
  searchContent:any;
  dialogRef: any = null;
  tabList = [{name: "Install", routerLink: "/install", preRoute: "install", hidden:false},
             {name: "Tutorials", routerLink: "/tutorials/overview", preRoute: "tutorials", hidden:false},
             {name: "Examples", routerLink: "/examples/overview", preRoute: "examples", hidden:false},
             {name: "API", routerLink: "/api/fe/Estimator", preRoute: "api", hidden:false},
             {name: "Community", routerLink: "/community", preRoute: "community", hidden:false}]

  @ViewChildren('tabDOM') tabDOMs: QueryList<ElementRef>;
  @ViewChild('logoDOM', {static:true}) logoDOM: ElementRef;
  @ViewChild('moreDOM', {read:ElementRef, static:true}) moreDOM: ElementRef;
  @ViewChild('searchDOM', {read:ElementRef, static:true}) searchDOM: ElementRef;
  tabBreakList:number[] = new Array(this.tabList.length);
  firstTabHideIndex:number;
  beforeMeasure = true;
  isSearchExpanded = false;
  searchBreak:number;

  structureHeaderDict = {
    'Content-Type': 'application/json',
    'Accept': "application/json, text/plain",
    'Access-Control-Allow-Origin': '*'
  }

  structureRequestOptions = {
    headers: new HttpHeaders(this.structureHeaderDict),
  };

  @HostBinding('class.loading')
  loading = false;

  minWidth: number = 1200;
  screenWidth: number;
  private screenWidth$ = new BehaviorSubject<number>(window.innerWidth);
  
  @HostListener('window:resize', ['$event'])
  onResize(event) {
    this.screenWidth$.next(event.target.innerWidth);
    this.checkBreaking();
    this.dealBreaking();
  }

  constructor(private router: Router,
              private http: HttpClient,
              public dialog: MatDialog,
              private globalService: GlobalService,
              private cd: ChangeDetectorRef) {
    this.screenWidth$.subscribe(width => {
      this.screenWidth = width;
    });
  }

  ngOnInit() {
    this.router.events.subscribe((val) => {
      if (val instanceof NavigationStart) {
        const ns = <NavigationStart>val;
        this.selected = ns.url.substring(1).split("/")[0];
      }
    });

    this.globalService.change.subscribe(loading => {
      this.loading = loading;
    });

    this.screenWidth$.subscribe(width => {
      this.screenWidth = width;
    });
  }

  ngAfterViewInit(){
    // measure the navbar tab length and get the breaking points
    this.getBreakPoint();
    this.beforeMeasure = false;
    this.checkBreaking();
    this.dealBreaking();
    this.cd.detectChanges();
  }

  preRoute(newSelection: string) {
    this.isNavbarCollapsed = !this.isNavbarCollapsed;
    this.selected = newSelection.toLowerCase();
  }

  search(content){
    var httpPrefix = "https://www.googleapis.com/customsearch/v1?q=";
    var httpPostfix = "&cx=008491496338527180074:d9p4ksqgel2&key=AIzaSyBLYeHKwpAOftKnYDsBAd4rSmX3VD9EJ7U";

    this.http.get(httpPrefix + content + httpPostfix, this.structureRequestOptions).subscribe(data => {
      if(this.dialogRef != null){
        this.dialog.closeAll();
      }
      this.dialogRef = this.dialog.open(DialogComponent, {
        minWidth:'50%',
        data: data
      });
    })
  }

  getBreakPoint(){
    var tabArray = this.tabDOMs.toArray();
    this.tabBreakList[0] = this.logoDOM.nativeElement.offsetWidth +
                           this.moreDOM.nativeElement.offsetWidth + 
                           tabArray[0].nativeElement.offsetWidth;

    for (var i=1;i<tabArray.length;i++){
      this.tabBreakList[i] = this.tabBreakList[i-1] + tabArray[i].nativeElement.offsetWidth;
    }

    this.searchBreak = this.searchDOM.nativeElement.offsetWidth + this.tabBreakList[tabArray.length-1]; 
  }

  checkBreaking(){
    for(var i=0;i<this.tabBreakList.length;i++){
      if(this.screenWidth < this.tabBreakList[i]){
        this.firstTabHideIndex = i;
        return;
      }
      this.firstTabHideIndex = this.tabBreakList.length;
    }
  }

  dealBreaking(){
    for(var i=0;i<this.tabList.length;i++){
      if( i < this.firstTabHideIndex){
        this.tabList[i].hidden = false;
      }
      else{
        this.tabList[i].hidden = true;
      }
    }
    this.moreDOM.nativeElement.hidden = this.getMoreHiddenBool() || this.isSearchExpanded;
  }

  checkAndDealSearchBreaking(){
    if(this.screenWidth < this.searchBreak){

    }
  }  

  getMoreHiddenBool(){
    if(this.beforeMeasure){
      return false;
    }

    for(var i=0;i<this.tabList.length;i++){
      if(this.tabList[i].hidden == true){
        return false;
      }
    }
    return true;
  }

  // onFocus(){
  //   var tmp = this.searchDOM.nativeElement;
  //   for(var i=0;i<15;i++){
  //     tmp = tmp.children[0];
  //   }
  //   console.log(this.searchDOM.nativeElement);
  //   console.log(tmp);
  //   console.log(tmp.attributes);
  // }

  onClick(){
    this.isSearchExpanded = !this.isSearchExpanded;
  }
}
