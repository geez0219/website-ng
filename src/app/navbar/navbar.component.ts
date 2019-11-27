import { Component, OnInit, Input, HostBinding } from '@angular/core';
import { NavigationStart, Router } from '@angular/router';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { MatDialog} from '@angular/material/dialog';
import { DialogComponent} from '../dialog/dialog.component'
import { GlobalService } from '../global.service';

@Component({
  selector: 'app-navbar',
  templateUrl: './navbar.component.html',
  styleUrls: ['./navbar.component.css']
})
export class NavbarComponent implements OnInit {
  isNavbarCollapsed=true;
  selected: string;
  searchContent:any
  dialogRef: any = null;
  
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

  constructor(private router: Router, 
              private http: HttpClient, 
              public dialog: MatDialog,
              private globalService: GlobalService) {
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
}
