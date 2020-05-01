import { Component, OnInit,
  HostBinding, HostListener, ElementRef, QueryList, ViewChildren, ViewChild, AfterViewInit, ChangeDetectorRef, Inject} from '@angular/core';
import { NavigationStart, Router } from '@angular/router';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { MatDialog} from '@angular/material/dialog';
import { DialogComponent} from '../dialog/dialog.component'
import { GlobalService } from '../global.service';
import { BehaviorSubject } from 'rxjs';
import { DOCUMENT } from '@angular/common'

@Component({
  selector: 'app-navbar',
  templateUrl: './navbar.component.html',
  styleUrls: ['./navbar.component.css']
})

export class NavbarComponent implements OnInit, AfterViewInit {
  isNavbarCollapsed = true;
  isMathidden = true;
  selected: string;
  searchContent: any;
  dialogRef: any = null;
  tabList = [{name: "Install", routerLink: "/install", preRoute: "install", hidden:false},
             {name: "Tutorials", routerLink: "/tutorials/beg/t01_getting_started", preRoute: "tutorials", hidden:false},
             {name: "Examples", routerLink: "/examples/image_classification/mnist", preRoute: "examples", hidden:false},
             {name: "API", routerLink: "/api/fe/Estimator", preRoute: "api", hidden:false},
             {name: "Community", routerLink: "/community", preRoute: "community", hidden:false}]

  @ViewChildren('tabDOM') tabDOMs: QueryList<ElementRef>;
  @ViewChild('logoDOM', {static:true}) logoDOM: ElementRef;
  @ViewChild('moreDOM', {read:ElementRef, static:true}) moreDOM: ElementRef;
  @ViewChild('searchDOM', {read:ElementRef, static:true}) searchDOM: ElementRef;
  @ViewChild('searchIconDOM', {static:true}) searchIconDOM: ElementRef;
  @ViewChild('inputDOM', {static:true}) inputDOM:ElementRef;

  tabBreakList:number[] = new Array(this.tabList.length);
  firstTabHideIndex:number;
  isMoreHidden = false;
  isSearchHidden = false;
  isSearchIconHidden = false;
  isSearchExpanded = false;
  errorPixel:number = 25;
  searchBreak:number;
  searchInMoreBreak:number;
  searchbarMinWidth:number = 200;

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
    this.checkAndDealSearchBreaking();
  }

  constructor(private router: Router,
              private http: HttpClient,
              public dialog: MatDialog,
              private globalService: GlobalService,
              @Inject(DOCUMENT) private _document,
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
    this.checkBreaking();
    this.dealBreaking();
    this.checkAndDealSearchBreaking();
    this.cd.detectChanges();
  }

  preRoute(newSelection: string) {
    this.isNavbarCollapsed = !this.isNavbarCollapsed;
    this.selected = newSelection.toLowerCase();
  }

  getBreakPoint(){
    var tabArray = this.tabDOMs.toArray();
    this.tabBreakList[0] = this.logoDOM.nativeElement.offsetWidth +
                           this.moreDOM.nativeElement.offsetWidth +
                           this.searchIconDOM.nativeElement.offsetWidth +
                           this.errorPixel +
                           tabArray[0].nativeElement.offsetWidth;


    for (var i=1;i<tabArray.length;i++){
      this.tabBreakList[i] = this.tabBreakList[i-1] + tabArray[i].nativeElement.offsetWidth;
    }

    this.tabBreakList[tabArray.length-1] = this.tabBreakList[tabArray.length-1] - this.moreDOM.nativeElement.offsetWidth;
    this.searchBreak = this.searchbarMinWidth + this.tabBreakList[tabArray.length-1] + this.searchIconDOM.nativeElement.offsetWidth;
    this.searchInMoreBreak = this.logoDOM.nativeElement.offsetWidth + this.moreDOM.nativeElement.offsetWidth + this.searchIconDOM.nativeElement.offsetWidth
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

    this.isMoreHidden = this.getMoreHiddenBool();
  }

  checkAndDealSearchBreaking(){
    if(this.screenWidth > this.searchBreak){
      this.isSearchHidden = false;
      this.isSearchIconHidden = true;
    }
    else if(this.screenWidth > this.searchInMoreBreak){
      this.isSearchHidden = true;
      this.isSearchIconHidden = false;
    }

    else{
      this.isSearchHidden = true;
      this.isSearchIconHidden = true;
    }
  }

  getMoreHiddenBool(){
    for(var i=0;i<this.tabList.length;i++){
      if(this.tabList[i].hidden == true){
        return false;
      }
    }
    return true;
  }

  expandSearch(){
    this.isSearchExpanded = true;
  }

  closeSearch(){
    this.isSearchExpanded = false;
  }
}
